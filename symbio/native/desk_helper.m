// The desk: a display that only Symbio works on, owned by this process.
//
// macOS has a private CoreGraphics class, CGVirtualDisplay, that adds a
// display with no panel behind it. The window server lays it out, draws on it
// and captures it like any monitor, so Symbio can open its windows there and
// leave the screen, pointer and keyboard in front of the user alone.
//
// This process exists only to own that display. It creates it, writes one
// line of JSON naming it, and waits: the display lives exactly as long as this
// process does. SIGTERM, SIGINT or SIGHUP hands it back to the window server.
//
//     symbio-desk --state PATH [--width 1440] [--height 900] [--name NAME]
//
// Everything else -- where the display sits, which windows go on it, looking
// at it and acting in it -- is symbio/desk.py. Policy belongs where it can be
// read and tested; this file is only the part that needs native code.
//
// The interfaces are declared by hand because the class is private. They
// match the selectors and type encodings the runtime reports on macOS 26
// (unsigned int sizes, a CGSize for the physical size, double refresh rate).
//
// Measured 2026-09-30 on macOS 26.5: while the Mac's own panel is asleep the
// window server defers every display unplug ("Deferred hotplugs cannot be
// processed in a non-wake state"), so a released desk stays online until the
// panel wakes. A second desk with the same serial is refused meanwhile ("no
// displayID available"), which is why the serial is chosen, not fixed.

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

@interface CGVirtualDisplayMode : NSObject
- (instancetype)initWithWidth:(unsigned int)width
                       height:(unsigned int)height
                  refreshRate:(double)refreshRate;
@end

@interface CGVirtualDisplaySettings : NSObject
@property(retain, nonatomic) NSArray *modes;
@property(nonatomic) unsigned int hiDPI;
@end

@interface CGVirtualDisplayDescriptor : NSObject
@property(retain, nonatomic) NSString *name;
@property(nonatomic) unsigned int maxPixelsWide;
@property(nonatomic) unsigned int maxPixelsHigh;
@property(nonatomic) CGSize sizeInMillimeters;
@property(nonatomic) unsigned int productID;
@property(nonatomic) unsigned int vendorID;
@property(nonatomic) unsigned int serialNum;
- (void)setDispatchQueue:(dispatch_queue_t)queue;
- (void)setTerminationHandler:(void (^)(id, id))handler;
@end

@interface CGVirtualDisplay : NSObject
- (instancetype)initWithDescriptor:(CGVirtualDisplayDescriptor *)descriptor;
- (BOOL)applySettings:(CGVirtualDisplaySettings *)settings;
@property(readonly, nonatomic) CGDirectDisplayID displayID;
@end

// How symbio/desk.py tells a desk from a real monitor: CGDisplayVendorNumber
// and CGDisplayModelNumber read these back.
static const unsigned int kVendor = 0x5359;
static const unsigned int kProduct = 0x4442;
static const unsigned int kMaxSerial = 32;

static CGVirtualDisplay *gDisplay;
static NSString *gStatePath;
static NSMutableArray *gSignalSources;

static void emit(NSDictionary *state, BOOL toStateFile) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:state options:0 error:nil];
    if (!json) return;
    if (toStateFile && gStatePath) {
        // Written beside and renamed over, so a reader never sees half a file.
        NSString *partial = [gStatePath stringByAppendingString:@".partial"];
        if ([json writeToFile:partial atomically:NO]) {
            rename(partial.fileSystemRepresentation, gStatePath.fileSystemRepresentation);
        }
    }
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

static void fail(NSString *why) {
    emit(@{@"ok" : @NO, @"pid" : @(getpid()), @"error" : why}, NO);
    exit(1);
}

static void releaseAndExit(int code) {
    // Only this process's state file is removed: a newer helper may already
    // have written its own over it.
    if (gStatePath) {
        NSData *data = [NSData dataWithContentsOfFile:gStatePath];
        NSDictionary *state =
            data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
        if ([state isKindOfClass:[NSDictionary class]] &&
            [state[@"pid"] intValue] == getpid()) {
            unlink(gStatePath.fileSystemRepresentation);
        }
    }
    // Dropping the last reference is what hands the display back.
    gDisplay = nil;
    exit(code);
}

static void onSignal(int sig) {
    signal(sig, SIG_IGN);
    dispatch_source_t source =
        dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, (uintptr_t)sig, 0,
                               dispatch_get_main_queue());
    dispatch_source_set_event_handler(source, ^{
      releaseAndExit(0);
    });
    dispatch_resume(source);
    [gSignalSources addObject:source];
}

// Serials already on a desk-looking display: an older desk whose unplug is
// still deferred, or another helper's. Reusing one is refused by the server.
static NSSet<NSNumber *> *serialsInUse(void) {
    CGDirectDisplayID ids[64];
    uint32_t count = 0;
    NSMutableSet *used = [NSMutableSet set];
    if (CGGetOnlineDisplayList(64, ids, &count) != kCGErrorSuccess) return used;
    for (uint32_t i = 0; i < count; i++) {
        if (CGDisplayVendorNumber(ids[i]) == kVendor &&
            CGDisplayModelNumber(ids[i]) == kProduct) {
            [used addObject:@(CGDisplaySerialNumber(ids[i]))];
        }
    }
    return used;
}

static BOOL isOnline(CGDirectDisplayID display) {
    CGDirectDisplayID ids[64];
    uint32_t count = 0;
    if (CGGetOnlineDisplayList(64, ids, &count) != kCGErrorSuccess) return NO;
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] == display) return YES;
    }
    return NO;
}

int main(int argc, const char *argv[]) {
    unsigned int width = 1440, height = 900, serial = 0;
    @autoreleasepool {
        NSString *name = @"Symbio Desk";
        for (int i = 1; i < argc; i++) {
            NSString *arg = @(argv[i]);
            NSString *value = i + 1 < argc ? @(argv[i + 1]) : nil;
            if (!value) break;
            if ([arg isEqualToString:@"--state"]) {
                gStatePath = value;
            } else if ([arg isEqualToString:@"--width"]) {
                width = (unsigned int)value.intValue;
            } else if ([arg isEqualToString:@"--height"]) {
                height = (unsigned int)value.intValue;
            } else if ([arg isEqualToString:@"--name"]) {
                name = value;
            } else {
                continue;
            }
            i++;
        }
        if (width < 640 || height < 480 || width > 3840 || height > 2160) {
            fail([NSString stringWithFormat:@"a desk of %ux%u is outside 640x480 to 3840x2160",
                                            width, height]);
        }

        gSignalSources = [NSMutableArray array];
        onSignal(SIGTERM);
        onSignal(SIGINT);
        onSignal(SIGHUP);

        NSSet *used = serialsInUse();
        for (unsigned int candidate = 1; candidate <= kMaxSerial && !gDisplay; candidate++) {
            if ([used containsObject:@(candidate)]) continue;
            CGVirtualDisplayDescriptor *descriptor = [[CGVirtualDisplayDescriptor alloc] init];
            [descriptor setDispatchQueue:dispatch_get_main_queue()];
            // The window server ends the display itself when it restarts. A
            // helper holding nothing must not look alive to desk.py.
            [descriptor setTerminationHandler:^(__unused id display, __unused id reason) {
              dispatch_async(dispatch_get_main_queue(), ^{
                releaseAndExit(3);
              });
            }];
            descriptor.name = name;
            descriptor.maxPixelsWide = width;
            descriptor.maxPixelsHigh = height;
            // ~96 dpi: an ordinary desktop monitor, so macOS offers the one
            // mode below at 1x rather than inventing a Retina "looks like".
            descriptor.sizeInMillimeters = CGSizeMake(width * 25.4 / 96.0, height * 25.4 / 96.0);
            descriptor.vendorID = kVendor;
            descriptor.productID = kProduct;
            descriptor.serialNum = candidate;
            gDisplay = [[CGVirtualDisplay alloc] initWithDescriptor:descriptor];
            if (gDisplay) serial = candidate;
        }
        if (!gDisplay) {
            fail(@"macOS would not create a virtual display (CGVirtualDisplay returned nil "
                 @"for every free serial)");
        }

        // One mode at 1x: a screenshot pixel is then a point, so a coordinate
        // read off a capture needs no Retina scale undone before it is used.
        CGVirtualDisplaySettings *settings = [[CGVirtualDisplaySettings alloc] init];
        settings.hiDPI = 0;
        settings.modes = @[ [[CGVirtualDisplayMode alloc] initWithWidth:width
                                                                 height:height
                                                            refreshRate:60] ];
        if (![gDisplay applySettings:settings]) {
            gDisplay = nil;
            fail(@"macOS refused the desk's display mode");
        }

        CGDirectDisplayID display = gDisplay.displayID;
        for (int waited = 0; waited < 60 && !isOnline(display); waited++) {
            usleep(50 * 1000);
        }
        if (!isOnline(display)) {
            gDisplay = nil;
            fail(@"the desk was created but never came online");
        }

        emit(@{
            @"ok" : @YES,
            @"pid" : @(getpid()),
            @"display" : @(display),
            @"serial" : @(serial),
            @"vendor" : @(kVendor),
            @"product" : @(kProduct),
            @"width" : @(width),
            @"height" : @(height),
            @"name" : name,
            @"started" : @((long long)[[NSDate date] timeIntervalSince1970]),
        },
             YES);
    }
    dispatch_main();
}
